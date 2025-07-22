
import uuid

from datetime import datetime
from typing import Dict, List, Optional

from foodie.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Foodie Card")
printer = PrettyPrinter

class FoodieCard:
    """
    Handles the creation and management of restaurant/seller cards for:
    - index.html (featured items)
    - restaurants.html (restaurant listings)
    - indie.html (independent sellers)
    """
    def __init__(self):
        self.config = load_global_config()
        self.required_fields = self.config.get('required_fields', [])

        self.card_config = get_config_section('foodie_card')
        self.max_title_length = self.card_config.get('max_title_length')
        self.max_description_length = self.card_config.get('max_description_length')
        self.allowed_card_types = self.card_config.get('allowed_card_types', [])
        self.default_image = self.card_config.get('default_image')
        self.rating_icon = self.card_config.get('rating_icon')

    def generate_cards_only(self, cards_data: List[Dict]) -> str:
        """Generates only card HTML without container"""
        return ''.join([self.generate_card_html(card) for card in cards_data])

    def generate_card_html(self, card_data: Dict) -> str:
        """
        Generates HTML for a foodie card based on card type and data
        
        Args:
            card_data: Dictionary containing card information
            
        Returns:
            str: HTML string for the card
        """
        # Validate input data
        self.validate_card_data(card_data)
        
        # Set defaults
        card_type = card_data.get('card_type', 'featured')
        card_id = str(uuid.uuid4())
        image_url = card_data.get('image_url', self.default_image)
        alt_text = card_data.get('alt_text', 'Foodie card image')
        title = card_data.get('title', '')
        description = card_data.get('description', '')
        rating = card_data.get('rating', 0)
        review_count = card_data.get('review_count', 0)
        link_text = card_data.get('link_text', 'View Details')
        link_url = card_data.get('link_url', '#')
        
        # Format rating display
        rating_display = ''
        if rating > 0:
            review_suffix = '+' if review_count > 0 else ''
            rating_display = (
                f'<span class="rating">'
                f'{self.card_config["rating_icon"]} '
                f'{rating} ({review_count}{review_suffix})'
                f'</span>'
            )
        
        # Card type specific adjustments
        if card_type == 'featured':
            link_text = card_data.get('link_text', 'Order Now')
        elif card_type == 'restaurant':
            link_text = card_data.get('link_text', 'View Menu')
        elif card_type == 'indie':
            link_text = card_data.get('link_text', 'View Offerings')
        
        # Generate HTML
        html = f"""
        <div class="card" id="{card_id}">
            <img src="{image_url}" alt="{alt_text}">
            <div class="card-content">
                <h3>{title}</h3>
                <p>{description}</p>
                {rating_display}
                <a href="{link_url}" class="card-link">{link_text}</a>
            </div>
        </div>
        """
        
        return html

    def validate_card_data(self, card_data: Dict) -> None:
        """Validate card data against business rules"""
        errors = []
        
        # Required fields validation
        required_fields = ['card_type', 'title', 'description', 'image_url']
        for field in required_fields:
            if field not in card_data:
                errors.append(f"Missing required field: {field}")
            elif not card_data[field]:
                errors.append(f"Field cannot be empty: {field}")
                
        # Card type validation
        if 'card_type' in card_data and card_data['card_type'] not in self.allowed_card_types:
            errors.append(
                f"Invalid card type: {card_data['card_type']}. "
                f"Allowed types: {', '.join(self.allowed_card_types)}"
            )
            
        # Length validation
        if 'title' in card_data and len(card_data['title']) > self.max_title_length:
            errors.append(f"Title exceeds {self.max_title_length} characters")
            
        if 'description' in card_data and len(card_data['description']) > self.max_description_length:
            errors.append(f"Description exceeds {self.max_description_length} characters")
            
        # Rating validation
        if 'rating' in card_data:
            rating = card_data['rating']
            if not (0 <= rating <= 5):
                errors.append("Rating must be between 0 and 5")
                
        if errors:
            raise ValueError("; ".join(errors))

    def generate_card_container(
        self, 
        cards_data: List[Dict], 
        container_class: str = "featured-grid",
        header: Optional[str] = None,
        subheader: Optional[str] = None
    ) -> str:
        """
        Generates a container with multiple cards
        
        Args:
            cards_data: List of card data dictionaries
            container_class: CSS class for the container
            header: Optional header text
            subheader: Optional subheader text
            
        Returns:
            str: HTML string for the card container
        """
        cards_html = ''.join([self.generate_card_html(card) for card in cards_data])
        
        header_html = ""
        if header:
            subheader_html = f'<p>{subheader}</p>' if subheader else ''
            header_html = f"""
            <div class="page-header">
                <h1>{header}</h1>
                {subheader_html}
            </div>
            """
        
        html = f"""
        <div class="container">
            {header_html}
            <div class="{container_class}">
                {cards_html}
            </div>
        </div>
        """
        
        return html
    
    def generate_dynamic_cards(
        self, 
        card_type: str,
        items: List[Dict],
        link_base_url: str = "#"
    ) -> List[Dict]:
        """
        Creates standardized card data from raw items
        
        Args:
            card_type: Type of cards to generate
            items: List of raw items (restaurants, sellers, or dishes)
            link_base_url: Base URL for card links
            
        Returns:
            List: Processed card data dictionaries
        """
        processed_cards = []
        
        for item in items:
            card_data = {
                'card_type': card_type,
                'image_url': item.get('image_url', self.default_image),
                'alt_text': item.get('alt_text', ''),
                'title': item.get('name', 'Untitled'),
                'description': item.get('description', ''),
                'rating': item.get('rating', 0),
                'review_count': item.get('review_count', 0),
                'link_url': f"{link_base_url}?id={item.get('id', '')}",
            }
            
            # Add type-specific fields
            if card_type == 'restaurant':
                card_data['link_text'] = 'View Menu'
            elif card_type == 'indie':
                card_data['link_text'] = 'View Offerings'
            elif card_type == 'featured':
                card_data['link_text'] = item.get('link_text', 'Order Now')
            
            processed_cards.append(card_data)
        
        return processed_cards

# Example usage
if __name__ == "__main__":
    card_builder = FoodieCard()
    
    # Create sample restaurant cards
    restaurants = [
        {
            'id': 'resto1',
            'name': "Gianni's Ristorante Italiano",
            'description': 'Authentic Italian pasta and seafood.',
            'rating': 4.8,
            'review_count': 250,
            'image_url': 'img/giannis.jpg',
            'alt_text': 'Italian restaurant'
        },
        {
            'id': 'resto2',
            'name': "Zeerovers",
            'description': 'Fresh catch-of-the-day, fried to perfection.',
            'rating': 4.9,
            'review_count': 500,
            'image_url': 'img/zeerovers.jpg',
            'alt_text': 'Seafood restaurant'
        }
    ]
    
    restaurant_cards = card_builder.generate_dynamic_cards(
        card_type='restaurant',
        items=restaurants,
        link_base_url='restaurant_details.html'
    )
    
    # Create sample indie seller cards
    indie_sellers = [
        {
            'id': 'indie1',
            'name': "Maria's Kitchen",
            'description': 'Authentic Aruban pastechi and stews.',
            'rating': 5.0,
            'review_count': 45,
            'image_url': 'img/marias_kitchen.jpg',
            'alt_text': 'Aruban home cooking'
        },
        {
            'id': 'indie2',
            'name': "The Cake Box",
            'description': 'Custom cakes, cupcakes, and desserts.',
            'rating': 4.9,
            'review_count': 80,
            'image_url': 'img/cake_box.jpg',
            'alt_text': 'Dessert shop'
        }
    ]
    
    indie_cards = card_builder.generate_dynamic_cards(
        card_type='indie',
        items=indie_sellers,
        link_base_url='indie_seller.html'
    )
    
    # Generate HTML containers
    restaurants_html = card_builder.generate_card_container(
        restaurant_cards,
        header="Explore Restaurants",
        subheader="Discover the best restaurants in Aruba, delivered to your doorstep."
    )
    
    indie_html = card_builder.generate_card_container(
        indie_cards,
        header="Taste the Talent: Foodie Indies",
        subheader="Support local entrepreneurs and discover unique, home-cooked meals."
    )
    
    print("Generated Restaurants Container:")
    print(restaurants_html)
    
    print("\nGenerated Indie Sellers Container:")
    print(indie_html)